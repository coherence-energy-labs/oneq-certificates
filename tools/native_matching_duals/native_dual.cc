// Research bridge over unchanged PyMatching Sparse Blossom source.
// Exports the terminal metric-matching dual BEFORE normal match extraction frees it.
// PyMatching source is Apache-2.0; see the pinned upstream LICENSE and source manifest.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "pymatching/sparse_blossom/matcher/mwpm.h"

using Clock = std::chrono::steady_clock;
static int64_t ns(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
}
struct Term {std::vector<int64_t> support; int64_t value;};
struct Context {
    pm::Mwpm solver;
    std::string result;
    Context(pm::MatchingGraph g,pm::SearchGraph s): solver(pm::GraphFlooder(std::move(g)),pm::SearchFlooder(std::move(s))) {}
};
static thread_local std::string last_error;
extern "C" const char* nd_error() {return last_error.c_str();}

extern "C" void* nd_create(int64_t n, int64_t m, const int64_t* edges) {
    try {
        if(n<=0 || n>1000000 || m<0 || m>10000000) throw std::invalid_argument("graph limits");
        pm::MatchingGraph graph(n,0);
        pm::SearchGraph search(n);
        std::set<std::pair<int64_t,int64_t>> seen;
        for(int64_t i=0;i<m;i++) {
            auto u=edges[3*i], v=edges[3*i+1], w=edges[3*i+2];
            if(u<0 || u>=n || v< -1 || v>=n || u==v || w<0 || w>(1<<25)-2 || w%2)
                throw std::invalid_argument("expected simple nonnegative even-integer graph");
            auto key=std::minmax(u,v);
            if(!seen.insert(key).second) throw std::invalid_argument("duplicate graph edge");
            if(v==-1) {graph.add_boundary_edge(u,w,{});search.add_boundary_edge(u,w,{});}
            else {graph.add_edge(u,v,w,{});search.add_edge(u,v,w,{});}
        }
        return new Context(std::move(graph),std::move(search));
    } catch(const std::exception& e) {last_error=e.what(); return nullptr;}
}
extern "C" void nd_free(void* p) {delete static_cast<Context*>(p);}

extern "C" const char* nd_decode(void* p, int64_t k, const int64_t* detectors, int export_dual) {
    auto* c=static_cast<Context*>(p);
    try {
        auto& solver=c->solver;
        auto& nodes=solver.flooder.graph.nodes;
        if(k<0 || static_cast<size_t>(k)>nodes.size()) throw std::invalid_argument("syndrome length");
        std::set<int64_t> syndrome;
        for(int64_t i=0;i<k;i++) {
            auto d=detectors[i];
            if(d<0 || static_cast<size_t>(d)>=nodes.size() || !syndrome.insert(d).second)
                throw std::invalid_argument("invalid or duplicate syndrome detector");
        }
        if(!solver.flooder.queue.empty()) throw std::invalid_argument("solver not clean");
        solver.flooder.queue.cur_time=0;
        auto begin=Clock::now();
        for(auto d:syndrome) solver.create_detection_event(&nodes[d]);
        while(true) {
            auto event=solver.flooder.run_until_next_mwpm_notification();
            if(event.event_type==pm::NO_EVENT) break;
            solver.process_event(event);
        }
        if(solver.node_arena.allocated.size()!=solver.node_arena.available.size())
            throw std::invalid_argument("no perfect matching");
        auto solved=Clock::now();
        std::vector<Term> terms;
        if(export_dual) {
            std::map<pm::GraphFillRegion*,int64_t> leaves;
            std::set<pm::GraphFillRegion*> roots;
            for(auto d:syndrome) {
                leaves[nodes[d].region_that_arrived]=d;
                roots.insert(nodes[d].region_that_arrived_top);
            }
            std::function<std::vector<int64_t>(pm::GraphFillRegion*)> visit;
            visit=[&](pm::GraphFillRegion* region) {
                if(!region || !region->radius.is_frozen()) throw std::invalid_argument("unfrozen dual");
                std::vector<int64_t> support;
                if(region->blossom_children.empty()) support.push_back(leaves.at(region));
                else for(auto& child:region->blossom_children) {
                    auto sub=visit(child.region); support.insert(support.end(),sub.begin(),sub.end());
                }
                std::sort(support.begin(),support.end());
                terms.push_back(Term{support,region->radius.y_intercept()});
                return support;
            };
            for(auto* region:roots) visit(region);
            std::sort(terms.begin(),terms.end(),[](auto& a,auto& b){return a.support<b.support;});
        }
        auto exported=Clock::now();
        std::vector<pm::CompressedEdge> pairs;
        for(auto d:syndrome) if(nodes[d].region_that_arrived)
            solver.shatter_blossom_and_extract_match_edges(nodes[d].region_that_arrived_top,pairs);
        auto extracted=Clock::now();
        std::set<std::pair<int64_t,int64_t>> correction;
        for(const auto& pair:pairs) {
            auto u=pair.loc_from-&nodes[0];
            auto v=pair.loc_to?pair.loc_to-&nodes[0]:SIZE_MAX;
            solver.search_flooder.iter_edges_on_shortest_path_from_middle(u,v,[&](const pm::SearchGraphEdge& e) {
                int64_t a=e.detector_node-&solver.search_flooder.graph.nodes[0];
                auto* target=e.detector_node->neighbors[e.neighbor_index];
                int64_t b=target?target-&solver.search_flooder.graph.nodes[0]:-1;
                auto edge=std::minmax(a,b);
                auto found=correction.find(edge);
                if(found==correction.end())correction.insert(edge);else correction.erase(found);
            });
        }
        auto expanded=Clock::now();
        std::ostringstream out;
        out<<"{\"schema\":\"oneq.native_metric_dual.v1\",\"pairs\":[";
        for(size_t i=0;i<pairs.size();i++) {
            if(i)out<<',';
            out<<'['<<(pairs[i].loc_from-&nodes[0])<<','
               <<(pairs[i].loc_to?pairs[i].loc_to-&nodes[0]:-1)<<']';
        }
        out<<"],\"correction\":[";
        size_t ci=0;
        for(const auto& edge:correction) {if(ci++)out<<',';out<<'['<<edge.first<<','<<edge.second<<']';}
        out<<"],\"dual\":[";
        for(size_t i=0;i<terms.size();i++) {
            if(i)out<<',';
            out<<"{\"support\":[";
            for(size_t j=0;j<terms[i].support.size();j++) {if(j)out<<',';out<<terms[i].support[j];}
            out<<"],\"value\":"<<terms[i].value<<'}';
        }
        out<<"],\"native_solve_ns\":"<<ns(begin,solved)
           <<",\"native_snapshot_ns\":"<<ns(solved,exported)
           <<",\"native_extract_ns\":"<<ns(exported,extracted)
           <<",\"native_expand_ns\":"<<ns(extracted,expanded)
           <<",\"native_serialize_ns\":"<<ns(expanded,Clock::now())<<'}';
        c->result=out.str();
        return c->result.c_str();
    } catch(const std::exception& e) {c->solver.reset(); last_error=e.what();return nullptr;}
}
